{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape : (2184, 3)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Time</th>\n",
       "      <th>Load (kW)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>01.09.2018</td>\n",
       "      <td>00:00:00</td>\n",
       "      <td>5551.82208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>NaN</td>\n",
       "      <td>01:00:00</td>\n",
       "      <td>4983.17184</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>NaN</td>\n",
       "      <td>02:00:00</td>\n",
       "      <td>4888.39680</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>NaN</td>\n",
       "      <td>03:00:00</td>\n",
       "      <td>5072.95872</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>NaN</td>\n",
       "      <td>04:00:00</td>\n",
       "      <td>5196.25980</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date      Time   Load (kW)\n",
       "0  01.09.2018  00:00:00  5551.82208\n",
       "1         NaN  01:00:00  4983.17184\n",
       "2         NaN  02:00:00  4888.39680\n",
       "3         NaN  03:00:00  5072.95872\n",
       "4         NaN  04:00:00  5196.25980"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_excel(\".\\LoadData.xlsx\")\n",
    "print(f'Shape : {data.shape}')\n",
    "data.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "a=[]\n",
    "b=[]\n",
    "for i in range(1,data.shape[0]):\n",
    "    a.append(data[\"Load (kW)\"].iloc[i-1])\n",
    "    b.append(data[\"Load (kW)\"].iloc[i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Previous Hour</th>\n",
       "      <th>Present Hour</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5551.82208</td>\n",
       "      <td>4983.17184</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4983.17184</td>\n",
       "      <td>4888.39680</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4888.39680</td>\n",
       "      <td>5072.95872</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5072.95872</td>\n",
       "      <td>5196.25980</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5196.25980</td>\n",
       "      <td>5641.29720</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Previous Hour  Present Hour\n",
       "0     5551.82208    4983.17184\n",
       "1     4983.17184    4888.39680\n",
       "2     4888.39680    5072.95872\n",
       "3     5072.95872    5196.25980\n",
       "4     5196.25980    5641.29720"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.DataFrame({'Previous Hour' : a, 'Present Hour' : b})\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Previous Hour</th>\n",
       "      <th>Present Hour</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.397877</td>\n",
       "      <td>0.293800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.293800</td>\n",
       "      <td>0.276454</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.276454</td>\n",
       "      <td>0.310234</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.310234</td>\n",
       "      <td>0.332801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.332801</td>\n",
       "      <td>0.414254</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Previous Hour  Present Hour\n",
       "0       0.397877      0.293800\n",
       "1       0.293800      0.276454\n",
       "2       0.276454      0.310234\n",
       "3       0.310234      0.332801\n",
       "4       0.332801      0.414254"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "max_x = np.max(data['Previous Hour'])\n",
    "max_y = np.max(data['Present Hour'])\n",
    "\n",
    "min_x = np.min(data['Previous Hour'])\n",
    "min_y = np.min(data['Present Hour'])\n",
    "\n",
    "data['Previous Hour'] = (data['Previous Hour'] - min_x) / (max_x - min_x)\n",
    "data['Present Hour'] = (data['Present Hour'] - min_y) / (max_y - min_y)\n",
    "\n",
    "data.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_Train, X_Test, Y_Train, Y_Test = train_test_split(data[\"Previous Hour\"], data[\"Present Hour\"], test_size=0.1, random_state=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "import math\n",
    "m = np.random.uniform(-4, 4)\n",
    "c = np.random.uniform(-4, 4)\n",
    "n = 0.04\n",
    "e = 1E-7\n",
    "v = 0.7\n",
    "epochs = 700\n",
    "em2 = 0\n",
    "ec2 = 0\n",
    "\n",
    "for i in range(epochs):\n",
    "\n",
    "    for j in range(X_Train.shape[0]):\n",
    "        gm = -1 * (Y_Train.iloc[j] - m * X_Train.iloc[j] - c) * X_Train.iloc[j]\n",
    "\n",
    "        # Calculating Grad C\n",
    "        gc = -1 * (Y_Train.iloc[j] - m * X_Train.iloc[j] - c)\n",
    "\n",
    "        # Calculating updated values of gm2 and gc2\n",
    "        em2 = (v * em2) + (1-v) * (gm * gm)\n",
    "        ec2 = (v * ec2) + (1-v) * (gc * gc)\n",
    "\n",
    "        # Updating m and c values\n",
    "        m -= (n * gm) / math.sqrt(e + em2)\n",
    "        c -= (n * gc) / math.sqrt(e + ec2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "M value is 0.8925636442921766 and C value is : 0.004634821612165153\n"
     ]
    }
   ],
   "source": [
    "print(f'M value is {m} and C value is : {c}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_p = []\n",
    "test_p = []\n",
    "for i in range(X_Train.shape[0]):\n",
    "    train_p.append(m * X_Train.iloc[i] + c)\n",
    "\n",
    "for i in range(X_Test.shape[0]):\n",
    "    test_p.append(m * X_Test.iloc[i] + c)\n",
    "\n",
    "# Denormalization\n",
    "train_p = [i * (max_y - min_y) + min_y for i in train_p]\n",
    "test_p = [i * (max_y - min_y) + min_y for i in test_p]\n",
    "Y_Train = [i * (max_y - min_y) + min_y for i in Y_Train]\n",
    "Y_Test = [i * (max_y - min_y) + min_y for i in Y_Test]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Error :\n",
      "MAE : 492.3671618898641\n",
      "MSE : 525328.2836526543\n",
      "RMSE : 724.7953391493727\n",
      "Testing Error :\n",
      "MAE : 473.8868154390221\n",
      "MSE : 498662.60393577267\n",
      "RMSE : 706.1604661376709\n"
     ]
    }
   ],
   "source": [
    "# Error Calculation\n",
    "\n",
    "train_MSE = 0\n",
    "train_MAE = 0\n",
    "\n",
    "for i in range(len(Y_Train)):\n",
    "    train_MAE+=abs(train_p[i] - Y_Train[i])\n",
    "for i in range(len(Y_Train)):\n",
    "    train_MSE+=(train_p[i] - Y_Train[i]) ** 2\n",
    "\n",
    "train_MAE/=len(Y_Train)\n",
    "train_MSE/=len(Y_Train)\n",
    "train_RMSE = train_MSE ** 0.5\n",
    "\n",
    "print('Training Error :')\n",
    "print(f'MAE : {train_MAE}')\n",
    "print(f'MSE : {train_MSE}')\n",
    "print(f'RMSE : {train_RMSE}')\n",
    "\n",
    "test_MAE = 0\n",
    "test_MSE = 0\n",
    "\n",
    "for i in range(len(Y_Test)):\n",
    "    test_MAE += abs(test_p[i] - Y_Test[i])\n",
    "for i in range(len(Y_Test)):\n",
    "    test_MSE += (test_p[i] - Y_Test[i]) ** 2\n",
    "\n",
    "test_MAE /=len(Y_Test)\n",
    "test_MSE /=len(Y_Test)\n",
    "test_RMSE = test_MSE ** 0.5\n",
    "\n",
    "print('Testing Error :')\n",
    "print(f'MAE : {test_MAE}')\n",
    "print(f'MSE : {test_MSE}')\n",
    "print(f'RMSE : {test_RMSE}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted load at present hour is 5322.32670337247 \n"
     ]
    }
   ],
   "source": [
    "load = float(input('Enter the load at previous hour : '))\n",
    "load = (load - min_x) / (max_x - min_x)\n",
    "prediction = m * load + c\n",
    "prediction = (prediction * (max_y - min_y)) + min_y\n",
    "print(f'Predicted load at present hour is {prediction} ')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "2be5faf79681da6f2a61fdfdd5405d65d042280f7fba6178067603e3a2925119"
  },
  "kernelspec": {
   "display_name": "Python 3.10.1 64-bit",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.1"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
