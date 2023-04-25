import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
csv_file = '/home/liuyiche/Desktop/Github/eecs442/basketball_registration/yolov5/runs/train/exp2/results.csv'
data = pd.read_csv(csv_file)


def draw_loss_vs_val(data):
    # Extract the epoch
    epochs = data.iloc[:, 0]

    training_bounding_box_loss = data.iloc[:, 1]
    training_object_loss = data.iloc[:, 2]
    training_classification_loss = data.iloc[:, 3]
    training_total_loss = training_bounding_box_loss + training_object_loss + training_classification_loss

    validation_bounding_box_loss = data.iloc[:, 8]
    validation_object_loss = data.iloc[:, 9]
    validation_classification_loss = data.iloc[:, 10]
    validation_total_loss = validation_bounding_box_loss + \
        validation_object_loss + validation_classification_loss

    # Create a line plot
    # plt.plot(epochs, training_bounding_box_loss, linestyle='-',
    #          label="Bounding Box Loss", color='r')
    # plt.plot(epochs, training_object_loss, linestyle='-',
    #          label="Object Loss", color='b')
    # plt.plot(epochs, training_classification_loss, linestyle='-',
    #          label="Classification Loss", color='g')
    plt.plot(epochs, training_total_loss, linestyle='-',
             label="Training", color='r')
    plt.plot(epochs, validation_total_loss, linestyle='-',
             label="Validation", color='k')

    # Customize the plot
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig('loss.png', dpi=300)


def draw_metric(data):
    # Extract the epoch
    epochs = data.iloc[:, 0]

    precision = data.iloc[:, 4]
    recall = data.iloc[:, 5]
    mAP_05 = data.iloc[:, 6]
    # mAP_05_95 = data.iloc[:, 7]

    # Create a line plot
    # plt.plot(epochs, precision, linestyle='-', label="Precision", color='r')
    # plt.plot(epochs, recall, linestyle='-', label="Recall", color='r')
    # plt.plot(epochs, mAP_05, linestyle='-', label="mAP 0.5", color='k')
    # plt.plot(epochs, mAP_05_95, linestyle='-', label="mAP 0.5:0.95", color='k')

    # Plot the precision-recall curve
    plt.plot(recall, precision, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)

    # Save the plot
    plt.savefig('Precision_Recall_Curve.png', dpi=300)


if __name__ == "__main__":
    draw_loss_vs_val(data)
    plt.clf()
    draw_metric(data)
