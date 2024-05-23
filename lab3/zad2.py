def train(model, data, optimizer, criterion, args):
  model.train()
  for batch_num, batch in enumerate(data):
    model.zero_grad()
    # ...
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    # ...


def evaluate(model, data, criterion, args):
  model.eval()
  with torch.no_grad():
    for batch_num, batch in enumerate(data):
      # ...
      logits = model(x)
      loss = criterion(logits, y)
      # ...

def main(args):
  seed = args.seed
  np.random.seed(seed)
  torch.manual_seed(seed)

  train_dataset, valid_dataset, test_dataset = load_dataset(...)
  model = initialize_model(args, ...)

  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  for epoch in range(args.epochs):
    train(...)
    evaluate(...)