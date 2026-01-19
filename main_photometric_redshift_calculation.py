from preprocessing import processed_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor



X, y = processed_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.45, random_state=42)


def model_estimator(model_name, depth_or_neighbors):  
    
    if model_name == KNeighborsRegressor: 
        model = model_name(depth_or_neighbors)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(y_pred.shape)
        # print(X)

        figure, ax_main = plt.subplots(figsize =(9.5, 7.5))

        ax_main.scatter(y_test, y_pred, label=f"neighbors = {depth_or_neighbors}", color="black")
        ax_main.set_ylabel(" Predicted Photometric Redshift. ")
        ax_main.set_xlabel(" True spectroscopic Redshift(z_spec). ")
        ax_main.set_title(" Predicted photo_z against z_spec. ")
        ax_main.legend(loc="best")

        figure.tight_layout()
        plt.show()
    else: 
        model  = model_name(max_depth=depth_or_neighbors, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        important_features = model.feature_importances_
        print(y_pred.shape)
        # print(X)

        figure, ax_main = plt.subplots(1,2,figsize =(12.5, 7.5))

        ax_main[0].scatter(y_test, y_pred, label=f"Max_depth = {depth_or_neighbors} ", color="black")
        ax_main[0].set_ylabel(" Predicted Photometric Redshift. ")
        ax_main[0].set_xlabel(" True spectroscopic Redshift(z_spec). ")
        ax_main[0].set_title(" Predicted photo_z against z_spec. ")
        ax_main[0].legend(loc="best")

        ax_main[1].bar(np.arange(len(important_features)), important_features, color="black")
        ax_main[1].set_ylabel(" Feature Importance. ")
        ax_main[1].set_xlabel(" Feature index in X(fitted data). ")
        ax_main[1].set_title(" Plot of feature importances. ")

        figure.tight_layout()
        plt.show()





if __name__ == "__main__":
    depth_or_neighbors = 7
    model_estimator(DecisionTreeRegressor, depth_or_neighbors)
    model_estimator(RandomForestRegressor, depth_or_neighbors)
    model_estimator(KNeighborsRegressor, depth_or_neighbors)
    # model_estimator()
    # I'm yet to implement the SVC
    
    