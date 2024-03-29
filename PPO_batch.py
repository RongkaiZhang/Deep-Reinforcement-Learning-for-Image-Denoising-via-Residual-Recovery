import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from Loader_batch import *
import cv2
from Environment import Cusenv

#Hyperparameters for PPO
gamma         = 0.95 
lmbda         = 1
eps_clip      = 0.5
K_epoch       = 15 # Minibatch_size in PPO
Train_Folder = 'BSD68/gray/train/'
Test_Folder = 'BSD68/gray/test/'
Crop_Size = 70

class PPO(nn.Module):
    def __init__(self, device, batch_size):
        super(PPO, self).__init__()
        self.data = [] # data buffer is built in PPO class
        self.batch_size = batch_size # input data is [batch_size, channel, H, W]
        #Load DnCNN weights and bias for initialization
        net = torch.load('dncnn_25.pth')
        #The layers to build network (FCN is used here and can be replaced for different tasks)
        self.conv1   = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.conv2   = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=True, padding_mode='zeros')
        self.conv3   = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3, bias=True, padding_mode='zeros')
        self.conv4   = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=4, dilation=4, bias=True, padding_mode='zeros')
        self.conv5_pi   = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3, bias=True, padding_mode='zeros')
        self.conv6_pi   = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=True, padding_mode='zeros')
        self.conv7_pi   = nn.Conv2d(in_channels=64, out_channels=27, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.conv5_v   = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3, bias=True, padding_mode='zeros')
        self.conv6_v   = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=True, padding_mode='zeros')
        self.conv7_v   = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.relu = nn.ReLU(inplace= True)
        #Initialize the weights and bias
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2.weight.data = net['model.6.weight']
        self.conv2.bias.data = net['model.6.bias']
        self.conv3.weight.data = net['model.12.weight']
        self.conv3.bias.data = net['model.12.bias']
        self.conv4.weight.data = net['model.18.weight']
        self.conv4.bias.data = net['model.18.bias']
        self.conv5_pi.weight.data = net['model.24.weight']
        self.conv5_pi.bias.data = net['model.24.bias']
        self.conv6_pi.weight.data = net['model.30.weight']
        self.conv6_pi.bias.data = net['model.30.bias']
        nn.init.kaiming_normal_(self.conv7_pi.weight)
        self.conv5_v.weight.data = net['model.24.weight']
        self.conv5_v.bias.data = net['model.24.bias']
        self.conv6_v.weight.data = net['model.30.weight']
        self.conv6_v.bias.data = net['model.30.bias']
        nn.init.kaiming_normal_(self.conv7_v.weight)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.device = device
        self.train = True

    def pi(self, x, softmax_dim = 1):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5_pi(x))
        x = self.relu(self.conv6_pi(x))
        x = self.conv7_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5_v(x))
        x = self.relu(self.conv6_v(x))
        v = self.conv7_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):# Convert list of transactions into tensors
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst= [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.extend(s)
            a_lst.extend(a)
            r_lst.extend(r)
            s_prime_lst.extend(s_prime)
            prob_a_lst.extend(prob_a)
            done_mask = 0 if done else 1
            done_lst.extend([done_mask])


        s, a, r, s_prime, prob_a, done_mask = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst).to(self.device), \
                                          torch.tensor(r_lst, dtype=torch.float).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                           torch.tensor(prob_a_lst).to(self.device), torch.tensor(done_lst, dtype=torch.float).to(self.device)
        self.data = []
        a = a.unsqueeze(1)
        prob_a = prob_a.unsqueeze(1)
        #s = s.squeeze(1)
        #r = r.squeeze(1)
        #s_prime = s_prime.squeeze(1)


        return s, a, r, s_prime, prob_a, done_mask.repeat_interleave(self.batch_size) 

    def train_net(self):# Train the network
        s, a, r, s_prime, prob_a, done_mask = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * torch.mul(self.v(s_prime), done_mask.view(r.shape[0], 1, 1, 1))
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            # reorder the transactions by batch
            for time in range(int(len(delta) / self.batch_size) - 1, -1, -1): 
                advantage = gamma * lmbda * advantage + delta[time*self.batch_size:(time+1)*self.batch_size] 
                for adv in advantage[::-1]:
                    advantage_lst.insert(0,adv)
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            pi = self.pi(s)
            #If use entropy in loss
            #pi = pi+1e-13 #1e-13 for stable training
            #entropy = (- pi*torch.log(pi)).sum(1).mean()
            pi_a = pi.gather(1, a)
            ratio = (pi_a / prob_a)  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach()) #-0.01*entropy if use entropy
            #loss = -torch.min(torch.mean(surr1), torch.mean(surr2)) + F.smooth_l1_loss(self.v(s), td_target.detach()) #-0.01*entropy if use entropy
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
#Train and validate in main() 
def main():
    #Define hyperparameters for training
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    print_interval = 1
    test_interval = 1000
    learning_rate = 0.001
    save_interval = 3000
    #Setup env. Cusenv() is a customized enironemnt which takes in the current state S and outputs the new state S_prime, Reward and Done (for episodic task).
    env = Cusenv()
    #Use PPO model.
    model = PPO(device,batch_size).to(device)
    #Load all data from files directly
    training_data = load_images_from_folder(Train_Folder)
    train_data_size = len(training_data)
    test_data = load_images_from_folder(Test_Folder)
    #Start to train 30001 episodes
    i = 0
    for n_epi in range(30001):#
        score = 0.0
        #Data preprocessing (augmentationm, crop, resize to (B,C,H,W), normalize)
        data = training_data[i:i + batch_size]
        data = data_augment(data)
        data = data_crop(data, Crop_Size)
        data = np.array(data)
        data = data[:, np.newaxis, :, :]
        raw_x = data / 255
        #Generate noise 
        raw_n = np.random.normal(0, 25, raw_x.shape).astype(raw_x.dtype) / 255
        #Reset env (here is just add noise to image)
        s = env.reset(raw_x, raw_n)
        done = False
        t_info = 0
        #Rollout 
        while not done:
            prob = model.pi(torch.from_numpy(s).float().to(device)).detach().cpu()
            prob = prob.permute(0,2,3,1)
            m = Categorical(prob)
            a = m.sample()
            prob_a = torch.exp(m.log_prob(a)).clone().numpy()
            action = a.numpy()
            s_prime, r, done = env.step(action, t_info)
            #if done:
                    #print('Epoch', n_epi,'Image',n, 'process', t_info, 'steps')
            t_info += 1
            model.put_data((s, action, r, s_prime, prob_a, done)) #All (s, action, r, s_prime, prob_a, done) are saved as numpy for convenience
            s = s_prime
            score += r
        #Train model after one rollout with batch_size images.
        model.train_net()

        #Show score (for image denoising with sigma = 25, the max score (when all noise is removed) should be )
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.3f}".format(n_epi, np.mean(score)*255))
            score = 0.0
            
        #Validate the model
        if n_epi%test_interval==0 and n_epi!=0:
            test_result = 0
            test_id = 0
            img_id = 0
            input_psnr = 0
            for im in test_data:
                data = im[np.newaxis, np.newaxis, :, :]
                raw_x = data / 255
                raw_n = np.random.normal(0, 25, raw_x.shape).astype(raw_x.dtype) / 255
                s = env.reset(raw_x, raw_n)
                I = np.maximum(0, raw_x)
                I = np.minimum(1, I)
                N = np.maximum(0, raw_x + raw_n)
                N = np.minimum(1, N)
                I = (I[0] * 255+0.5).astype(np.uint8)
                N = (N[0] * 255+0.5).astype(np.uint8)
                I = np.transpose(I, (1, 2, 0))
                N = np.transpose(N, (1, 2, 0))
                cv2.imwrite('result_noentropy/' + str(test_id) + '_input.png', N)
                psnr1_cv = cv2.PSNR(N, I)
                #psnr1 = np.mean(10 * np.log10(1 / (s - env.ground_truth) ** 2))
                #for t in range(T_horizon):
                done = False
                t = 0
                while not done:
                    prob = model.pi(torch.from_numpy(s).float().to(device))
                    #prob = prob.permute(0,2,3,1).detach().cpu()
                    #m = Categorical(prob)
                    _, a = torch.max(prob, 1)
                    action = a.cpu().numpy()
                    s_prime, r, done = env.step(action,t)
                    if done:
                        print('test image', img_id, 'process', t, 'steps')
                    s = s_prime
                    t += 1
                img_id += 1
                p = np.maximum(0, s)
                p = np.minimum(1, p)
                p = (p[0] * 255+0.5).astype(np.uint8)
                p = np.transpose(p, (1, 2, 0))
                #p = cv2.blur(p,(3,3))
                cv2.imwrite('result_noentropy/' + str(test_id) + '_output.png', p)
                #psnr2 = np.mean(10*np.log10(1/(s-env.ground_truth)**2))
                psnr2_cv = cv2.PSNR(p, I)
                test_result += psnr2_cv
                print('test: PSNR_CV before:',psnr1_cv, 'PSNR_CV after:', psnr2_cv)
                test_id +=1
            print('Overall performance:', test_result/len(test_data))

       
        if n_epi%save_interval==0 and n_epi!=0:
            torch.save(model,'PPO_model_{}.pt'.format(n_epi))
            score = 0.0

        if i + batch_size >= train_data_size:
            i = 0

        else:
            i += batch_size

        if i + 2 * batch_size >= train_data_size:
            i = train_data_size - batch_size

        model.optimizer = optim.Adam(model.parameters(), lr=learning_rate*((1-n_epi/30001)**0.9)) #Lr decay for stable training

if __name__ == '__main__':
    main()
