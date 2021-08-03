# --------------------------------------------------------------------
# To compare with pioneers, we quoted the train procedure from EPIVAN
# --------------------------------------------------------------------
from models import get_model
import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import roc_auc_score,average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

class roc_callback(Callback):
    def __init__(self, val_data,name):
        self.en = val_data[0]
        self.pr = val_data[1]
        self.y = val_data[2]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.en,self.pr])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)
        self.model.save_weights("./model/%sModel%d.h5" % (self.name, epoch))
        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

name = 'GM12878'
Data_dir = './data1/%s/'%name
max_len_en = 3000
max_len_pr = 2000
nwords = 4097
emb_dim = 100

train = np.load(Data_dir+'%s_train.npz'% name)
test = np.load(Data_dir+'%s_test.npz'% name)
X_en_tra, X_pr_tra, y_tra = train['X_en_tra'], train['X_pr_tra'], train['y_tra']
X_en_tes, X_pr_tes, y_tes = test['X_en_tes'], test['X_pr_tes'], test['y_tes']


# mc = ModelCheckpoint('model/best_model_HUVEC.h5', save_best_only=True,
#                 save_weights_only=True)



X_en_tra, X_en_val, X_pr_tra, X_pr_val, y_tra, y_val = train_test_split(
    X_en_tra, X_pr_tra, y_tra, test_size=0.05,stratify=y_tra, random_state=250)


model = get_model(max_len_en, max_len_pr, nwords, emb_dim)
#model.load_weights("model/genModel.h5")
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
model.summary()

back = roc_callback(val_data=[X_en_val, X_pr_val, y_val], name=name)
history = model.fit([X_en_tra, X_pr_tra], y_tra,
                    validation_data=([X_en_val, X_pr_val], y_val),
                    epochs=15, batch_size=
                    64,
                    callbacks=[back]
                  )








