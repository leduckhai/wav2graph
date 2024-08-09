from sklearn.metrics import roc_auc_score, average_precision_score


# AUC score
def auc_score(out, y, mask):
    y = y[mask].cpu().numpy()
    out = out[mask].detach().cpu().numpy()
    return roc_auc_score(y, out)

# AP score
def ap_score(out, y, mask):
    y = y[mask].cpu().numpy()
    out = out[mask].detach().cpu().numpy()
    return average_precision_score(y, out)
