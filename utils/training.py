import torch
from tqdm import tqdm
from segmentation_models_pytorch.losses import DiceLoss
from config import *

def training_cycle(cfg, model, train_loader, val_loader, criterion, optimizer,
                       scheduler, early_stopping, model_name='cnn', out_dir=model_results, device='cpu', num_epochs=200):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    val_loss = 10 ** 16
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (inputs, masks) in enumerate(tepoch):

                inputs, masks = inputs.to(device), masks.to(device)

                optimizer.zero_grad()
                if not cfg.opt.logit_loss:
                    if cfg.model.activation != None:
                        outputs = model(inputs.to(device))
                    else:
                        outputs = model(inputs.to(device)).sigmoid()
                else:
                    outputs = model(inputs.to(device))

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

            # Print average training loss for the epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}")
            train_losses.append(running_loss / len(train_loader))

            # Validation loop
            model.eval()
            running_loss = 0.0

            with torch.no_grad():
                with tqdm(val_loader, unit="batch") as vepoch:
                    for i, (inputs, masks) in enumerate(vepoch):

                        inputs, masks = inputs.to(device), masks.to(device)

                        if not cfg.opt.logit_loss:
                            if cfg.model.activation != None:
                                outputs = model(inputs.to(device))
                            else:
                                outputs = model(inputs.to(device)).sigmoid()
                        else:
                            outputs = model(inputs.to(device))

                        loss += criterion(outputs, masks).item()
                        running_loss += loss

                        vepoch.set_postfix(loss=loss)

                val_loss_epoch = running_loss / len(val_loader)
                val_losses.append(val_loss_epoch)

                scheduler.step(val_loss_epoch)
                print('eval loss {}'.format(val_loss_epoch))
                early_stopping(val_loss_epoch)
                if early_stopping.early_stop:
                    break
                if val_loss_epoch < val_loss:
                    print('val_loss improved from {} to {}, saving model  {} to {}' \
                          .format(val_loss, val_loss_epoch, model_name, out_dir))
                    torch.save({
                        'cfg': cfg,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss_epoch,
                        'train_loss_history': train_losses,
                        'val_loss_history': val_losses,
                    }, out_dir + '/{}.pth'.format(model_name))
                    val_loss = val_loss_epoch


def training_cycle_deeplab(cfg, model, train_loader, val_loader, criterion, optimizer,
                       scheduler, early_stopping, model_name='cnn'
                           , out_dir=model_results, device='cpu', num_epochs=200):

    metric_dice_loss = DiceLoss(mode='binary', from_logits=cfg.opt.logit_loss)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    val_loss = 10 ** 16
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_dice_loss= 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (inputs, masks) in enumerate(tepoch):

                inputs, masks = inputs.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                if not cfg.opt.logit_loss:
                    loss = criterion(outputs['out'].sigmoid(), masks)
                    dice_loss = metric_dice_loss(outputs['out'].sigmoid(), masks)
                else:
                    loss = criterion(outputs['out'], masks)
                    dice_loss = metric_dice_loss(outputs['out'], masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_dice_loss += dice_loss.item()
                tepoch.set_postfix(loss=loss.item(), dice_loss=dice_loss.item())

            # Print average training loss for the epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)},"
                  f"Training Dice Loss: {running_dice_loss / len(train_loader)}")
            train_losses.append(running_loss / len(train_loader))

            # Validation loop
            model.eval()
            running_loss = 0.0
            running_dice_loss = 0.0

            with torch.no_grad():
                with tqdm(val_loader, unit="batch") as vepoch:
                    for i, (inputs, masks) in enumerate(vepoch):

                        inputs, masks = inputs.to(device), masks.to(device)

                        if not cfg.opt.logit_loss:
                            loss = criterion(outputs['out'].sigmoid(), masks)
                            dice_loss = metric_dice_loss(outputs['out'].sigmoid(), masks)
                        else:
                            loss = criterion(outputs['out'], masks)
                            dice_loss = metric_dice_loss(outputs['out'], masks)

                        loss = criterion(outputs, masks).item()

                        running_loss += loss

                        vepoch.set_postfix(loss=loss, dice_loss=dice_loss.item())

                val_loss_epoch = running_loss / len(val_loader)
                val_dice_loss_epoch = running_dice_loss / len(val_loader)
                val_losses.append(val_loss_epoch)

                scheduler.step(val_loss_epoch)
                print('eval loss {} Training Dice Loss: {}'.format(val_loss_epoch, val_dice_loss_epoch))

                early_stopping(val_loss_epoch)
                if early_stopping.early_stop:
                    break
                if val_loss_epoch < val_loss:
                    print('val_loss improved from {} to {}, saving model  {} to {}' \
                          .format(val_loss, val_loss_epoch, model_name, out_dir))
                    torch.save({
                        'cfg': cfg,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss_epoch,
                        'train_loss_history': train_losses,
                        'val_loss_history': val_losses,
                    }, out_dir + '/{}.pth'.format(model_name))
                    val_loss = val_loss_epoch
