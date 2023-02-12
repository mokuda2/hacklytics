import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib import style



columns = [['Ask Price'+str(i), 'Ask Volume'+str(i), 'Bid Price'+str(i), 'Bid Volume'+str(i)]for i in range(1,11)]
columns = [item for sublist in columns for item in sublist]
bid_ask = pd.read_csv('hacklytics\lobster\AMZN_2012-06-21_34200000_57600000_orderbook_10.csv' , header=None)
bid_ask.columns = columns
columns_bis = [['Ask Price'+str(i)]for i in range(1,11)] +[['Ask Volume'+str(i)]for i in range(1,11)] + [['Bid Price'+str(i)]for i in range(1,11)]  +  [['Bid Volume'+str(i)]for i in range(1,11)]
columns_bis = [item for sublist in columns_bis for item in sublist]
bid_ask = bid_ask[columns_bis]


ask_price = bid_ask[['Ask Price'+str(i) for i in range(1,11)]]
ask_price = ask_price.T
ask_price.reset_index(drop=True, inplace=True)

ask_volume = bid_ask[['Ask Volume'+str(i) for i in range(1,11)]]
ask_volume = ask_volume.T
ask_volume.reset_index(drop=True, inplace=True)


bid_price = bid_ask[['Bid Price'+str(i) for i in range(1,11)]]
bid_price = bid_price.T
bid_price.reset_index(drop=True, inplace=True)

bid_volume = bid_ask[['Bid Volume'+str(i) for i in range(1,11)]]
bid_volume = bid_volume.T
bid_volume.reset_index(drop=True, inplace=True)


ask = pd.concat([ask_price, ask_volume], axis=1)
bid = pd.concat([bid_price, bid_volume], axis=1)
ask.columns = range(ask.shape[1])
bid.columns = range(bid.shape[1])
_, a2 = ask_price.shape
columns_ask = [[0+i,a2+i] for i in range(a2)]
columns_ask = [item for sublist in columns_ask for item in sublist]
ask = ask[columns_ask]
#do the same for bid
columns_bid = [[0+i,a2+i] for i in range(a2)]
columns_bid = [item for sublist in columns_bid for item in sublist]
bid = bid[columns_bid]
data_ask_side = ['ask']*10
data_bid_side = ['bid']*10
#create dataframe with ask and bid
data_ask_side = pd.DataFrame(data_ask_side, columns=['side'])
data_ask_side.reset_index(drop=True, inplace=True)
data_bid_side = pd.DataFrame(data_bid_side, columns=['side'])
data_bid_side.reset_index(drop=True, inplace=True)
ask.reset_index(drop=True, inplace=True)
bid.reset_index(drop=True, inplace=True)
columns_final = [['price'+str(i), 'volume'+str(i)]for i in range(a2)]
columns_final = [item for sublist in columns_final for item in sublist]
ask.columns = columns_final
bid.columns = columns_final
ask['side'] = data_ask_side
bid['side'] = data_bid_side
ask.reset_index(drop=True, inplace=True)
bid.reset_index(drop=True, inplace=True)
all = pd.concat([ask, bid])



n,m = all.shape
prices = ['price'+str(i) for i in range(m//2)]
x_inf, x_sup = all[prices].quantile(0.25, axis=1)[9].values[1], all[prices].quantile(0.90, axis=1)[9].values[0]
volumes = ['volume'+str(i) for i in range(m//2)]
y_lim = (all[volumes][:10].sum().quantile(0.90) + all[volumes][10:].sum().quantile(0.90))/2

style.use('fivethirtyeight')

fig = plt.figure(figsize=(20,9))
ax = fig.add_subplot(1,1,1)

def animate(i):
    df1 = all[['price'+str(i), 'volume'+str(i), 'side']]
    ax.clear()
    sns.ecdfplot(x="price"+str(i), weights="volume"+str(i), stat="count", complementary=True, data=df1[10:], ax=ax, color='green')
    sns.ecdfplot(x="price"+str(i), weights="volume"+str(i), stat="count", data=df1[:10], ax=ax, color='red')
    sns.scatterplot(x="price"+str(i), y="volume"+str(i), hue="side", data=df1, ax=ax, palette=['red', 'green'])
    #define y label
    ax.set_ylabel('Volume')

    ax.axvline(x=df1['price'+str(i)][0].mean() , color='black', linestyle='--')


    ax.set_xlim(2.23e6, 2.25e6)
    ax.set_ylim(-100, y_lim)
    plt.title('Bid-Ask Spread Evolution for Amazon Stock')

ani = animation.FuncAnimation(fig, animate, interval=0.01)

plt.show()
