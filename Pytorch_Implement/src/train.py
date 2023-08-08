from functional import apply_model, evaluate, SaveBestModel, save_model, save_plots
import zero

import torch
#from torchmetrics.regression import MeanSquaredError

def train(model, optimizer, criterion, n_epochs, batch_size, X, y, checkpoint_path, device):

    train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)
    progress = zero.ProgressTracker(patience=15)
    task_type = 'regression'
    #mean_squared_error = MeanSquaredError()

    train_loss, valid_loss = [], []
    train_rmse, valid_rmse = [], []

    save_best_model = SaveBestModel(checkpoint_path = checkpoint_path)

    report_frequency = len(X['train']) // batch_size // 5
    for epoch in range(1, n_epochs + 1):

        train_running_loss = 0.0
        train_running_rmse = 0
        counter = 0

        for iteration, batch_idx in enumerate(train_loader):
            counter += 1
            model.train()
            optimizer.zero_grad()
            x_batch = X['train'][batch_idx]
            y_batch = y['train'][batch_idx]
            #loss = loss_fn(apply_model(x_batch).squeeze(1), y_batch)
            output = apply_model(device, model, x_batch).squeeze(1)
            loss = criterion(apply_model(device, model, x_batch).squeeze(1), y_batch)

            train_running_loss += loss.item()
            #train_mse = mean_squared_error(y_batch, output)
            #train_rmse_score = torch.sqrt(train_mse)

            #train_running_rmse += train_rmse_score

            loss.backward()
            optimizer.step()

            if iteration % report_frequency == 0:
                print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.5f}')

        # loss and accuracy for the complete epoch
        epoch_loss = train_running_loss / counter
        #epoch_mse = train_running_rmse / counter

        val_score, val_mse_loss = evaluate('training', model, optimizer, X, y, 'val', checkpoint_path, device)
        save_best_model(
            val_score, epoch, model, optimizer, criterion
        )

        train_loss.append(epoch_loss)
        valid_loss.append(val_mse_loss)
        train_rmse.append(0.1)
        valid_rmse.append(val_score)

        print('-'*50)

        #test_score = evaluate('test')
        print(f'Epoch {epoch:03d} | Validation score: {val_score}', end='')
        progress.update((-1 if task_type == 'regression' else 1) * val_score)
        if progress.success:
            print(' <<< BEST VALIDATION EPOCH', end='')
        print()
        if progress.fail:
            break

    # save the trained model weights for a final time
    
    # final_model_path = ''
    # save_model(n_epochs, model, optimizer, loss, final_model_path)
    # save the loss and accuracy plots
    save_plots(train_rmse, valid_rmse, train_loss, valid_loss)
    print('TRAINING COMPLETE')