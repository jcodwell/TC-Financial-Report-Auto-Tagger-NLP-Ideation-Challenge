from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
import torch
from transformers import AdamW
from tqdm.auto import tqdm 
from dataset import Dataset
from mlm import MLM
from torch.utils.tensorboard import SummaryWriter

config = RobertaConfig(
    vocab_size=15_000,  # we align this to the tokenizer vocab_size
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=3,
    type_vocab_size=1
)
model = RobertaForMaskedLM(config)


#Tensor Board Writer
writer = SummaryWriter('runs/experiment_2')


#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device

device = torch.device('cpu')


model.to(device)
# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-4)
mlm =MLM() 

D =  Dataset(mlm.create_tensor_attributes())

dataLoader = torch.utils.data.DataLoader(D, batch_size=1, shuffle=True )

#training Loops
epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(dataLoader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        model.save_pretrained('./tokenizer_model') 