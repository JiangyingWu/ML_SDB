import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import joblib  # For loading machine learning models
import spectral as sp  # For loading spectral data
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import random
from matplotlib.widgets import Slider


# 设置随机种子
def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)  # Python 内置随机库
    np.random.seed(seed)  # NumPy 随机库
    # torch.manual_seed(seed)  # PyTorch 随机库
    # torch.cuda.manual_seed(seed)  # 当前 CUDA 设备
    # torch.cuda.manual_seed_all(seed)  # 所有 CUDA 设备
    # torch.backends.cudnn.deterministic = True  # 确定性卷积操作   
    # torch.backends.cudnn.benchmark = False  # 禁止自动优化
setseed = 3407
set_random_seed(setseed)
# rows, cols 可以随机调节，1~100
rows, cols = 90, 90
start_row = random.randint(0, 100-rows)
start_col = random.randint(0, 100-cols)


# 1. Function to load spectral data
def load_spectral_data(hdr_path, dat_path):

    # 1.1. Load spectral data
    head = sp.envi.open(hdr_path, dat_path)
    print(head)
    
    data = head.load()
    print(f'original data shape: {data.shape}')

    # 1.2. Reshape data to 2D array
    m, n, ch = data.shape

    # 1.2.1.先全部reshape到100*100*ch，然后再截取
    if m > 0 and m < 100:
        # 补充到100行，从第m-1行开始往前复制
        data = np.vstack((data, np.tile(data[m-1, :, :], (100-m, 1))))
    elif m > 100:
        # 截取前100行
        data = data[:100, :, :]
    elif m == 100:
        pass
    else:
        return None

    if n > 0 and n < 100:
        # 补充到100列，从第n-1列开始往前复制
        data = np.hstack((data, np.tile(data[:, n-1, :], (1, 100-n))))
    elif n > 100:
        # 截取前100列
        data = data[:, :100, :]
    elif n == 100:
        pass
    else:
        return None

    # 1.2.2.再截取
    if m > 0 and m < rows:
        # 补充到rows行，从第m-1行开始往前复制
        data = np.vstack((data, np.tile(data[m-1, :, :], (rows-m, 1))))
    elif m > rows:
        # 截取随机rows行
        data = data[start_row:start_row+rows, :, :]
    elif m == rows:
        pass
    else:
        return None
    
    if n > 0 and n < cols:
        # 补充到cols列，从第n-1列开始往前复制
        data = np.hstack((data, np.tile(data[:, n-1, :], (1, cols-n))))
    elif n > cols:
        # 截取随机cols列
        data = data[:, start_col:start_col+cols, :]
    elif n == cols:
        pass
    else:
        return None
    
    
    data = data.reshape(-1, ch)
    print(f'reshaped data shape: {data.shape}')

    # 1.3. Normalize data
    datamin, datamax = np.min(data), np.max(data)
    data = (data - datamin) / (datamax - datamin)
    print(f'normalized data shape: {data.shape}, min: {datamin}, max: {datamax}')

    return data, datamin, datamax


# 2. Function to make predictions
def predict_with_model(data, model_path):
    model = joblib.load(model_path)
    print(f'Loaded model: {model_path}')
    if model_path != "./ckpt/xgboost.pkl":
        print(f'data shape: {data.shape}')
    predictions = model.predict(data)
    return predictions


# 3. Tkinter application
class SpectralApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Data Prediction GUI")
        self.root.state('zoomed')  # 设置窗口默认全屏

        # 创建主框架
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame = main_frame

        # 左侧框架（标题和按钮）
        left_frame = tk.Frame(main_frame, width=500, padx=10, pady=10, borderwidth=1, relief="solid")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.left_frame = left_frame

        # 右侧框架（图像输出）
        right_frame = tk.Frame(main_frame, padx=10, pady=10, borderwidth=1, relief="solid")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.right_frame = right_frame

        # 标题
        tk.Label(left_frame, text="Spectral Data Prediction GUI", font=("Helvetica", 20), pady=10).pack()
        
        # 按钮 - 选择光谱文件
        tk.Label(left_frame, text="Select spectral .hdr file & .dat file:", font=("Helvetica", 14), pady=10).pack()
        tk.Button(left_frame, text="Load HDR File", command=self.load_hdr, font=("Helvetica", 10)).pack(pady=5, fill=tk.X)
        tk.Button(left_frame, text="Load DAT File", command=self.load_dat, font=("Helvetica", 10)).pack(pady=5, fill=tk.X)
        
        # 按钮 - 选择深度文件
        tk.Label(left_frame, text="Select real depth .hdr file & .dat file:", font=("Helvetica", 14), pady=10).pack()
        tk.Button(left_frame, text="Load HDR File", command=self.load_depth_hdr, font=("Helvetica", 10)).pack(pady=5, fill=tk.X)
        tk.Button(left_frame, text="Load DAT File", command=self.load_depth_dat, font=("Helvetica", 10)).pack(pady=5, fill=tk.X)
        
        # 按钮 - 选择预测模型
        tk.Label(left_frame, text="Select a prediction model:", font=("Helvetica", 14), pady=10).pack()
        models = {
            "Random Forest": "./ckpt/random_forest.pkl",
            "Decision Tree": "./ckpt/decision_tree.pkl",
            "SVM": "./ckpt/svm.pkl",
            "PLS": "./ckpt/pls.pkl",
            "KNN": "./ckpt/knn.pkl",
            "XGBoost": "./ckpt/xgboost.pkl",
            "CATBoost": "./ckpt/catboost.pkl"
        }
        for model_name, model_file in models.items():
            tk.Button(self.left_frame, text=model_name, command=lambda m=model_file: self.run_prediction(m), font=("Helvetica", 10)).pack(pady=5, fill=tk.X)

        # 初始化占位图的 Canvas
        self.canvas = None
        fig, modelname = self.plot_placeholder()

        # 按钮 - 保存右侧图像
        tk.Button(self.left_frame, text="Save Plot", command=lambda: self.save_plot(fig, modelname), font=("Helvetica", 10)).pack(pady=5, fill=tk.X)


    # 3.1. Load hdr & dat files---------------------------------------------------------------------------------------------------------------------------------------
    # Function to load Spectral HDR file
    def load_hdr(self):
        file_path = filedialog.askopenfilename(filetypes=[("HDR Files", "*.hdr")])
        if file_path:
            self.hdr_file = file_path
            # messagebox.showinfo("File Loaded", f"HDR file loaded: {file_path}")
            print(f'Loaded HDR file: {file_path}')

    # Function to load Spectral DAT file
    def load_dat(self):
        file_path = filedialog.askopenfilename(filetypes=[("DAT Files", "*.dat")])
        if file_path:
            self.dat_file = file_path
            # messagebox.showinfo("File Loaded", f"DAT file loaded: {file_path}")
            print(f'Loaded DAT file: {file_path}')

    # Function to load Depth HDR file
    def load_depth_hdr(self):
        file_path = filedialog.askopenfilename(filetypes=[("HDR Files", "*.hdr")])
        if file_path:
            self.depth_hdr_file = file_path
            # messagebox.showinfo("File Loaded", f"Depth HDR file loaded: {file_path}")
            print(f'Loaded Depth HDR file: {file_path}')

    # Function to load Depth DAT file
    def load_depth_dat(self):
        file_path = filedialog.askopenfilename(filetypes=[("DAT Files", "*.dat")])
        if file_path:
            self.depth_dat_file = file_path
            # messagebox.showinfo("File Loaded", f"Depth DAT file loaded: {file_path}")
            print(f'Loaded Depth DAT file: {file_path}')


    # 3.2. Run prediction--------------------------------------------------------------------------------------------------------------------------------------------
    # Function to run prediction
    def run_prediction(self, model_file):
        if not self.hdr_file or not self.dat_file:
            messagebox.showerror("Error", "Please load both HDR and DAT files.")
            return
        
        # if not self.depth_hdr_file or not self.depth_dat_file:
        #     messagebox.showerror("Error", "Please load both Depth HDR and Depth DAT files.")
        #     return
        
        try:
            # Load spectral data
            spectral_data, datamin, datamax = load_spectral_data(self.hdr_file, self.dat_file)
            spectral_data4plot = spectral_data[:]

            # Example: Replace with your actual data preprocessing
            if spectral_data is None or spectral_data.shape[0] == 0:
                raise ValueError("Invalid spectral data.")
            
            # Predict using model
            if model_file == "./ckpt/xgboost.pkl":
                spectral_data = xgb.DMatrix(spectral_data)
            predicted = predict_with_model(spectral_data, model_file)
            print(f'Predicted shape: {predicted.shape}')
            
            if self.depth_hdr_file is not None and self.depth_dat_file is not None:
                # Load depth data
                actual_depth_data, depth_datamin, depth_datamax = load_spectral_data(self.depth_hdr_file, self.depth_dat_file)

                # Plot results
                self.plot_results(spectral_data4plot, actual_depth_data, predicted, depth_datamin, depth_datamax, model_file)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")



    # 3.3. Plot results-----------------------------------------------------------------------------------------------------------------------------------------------

    def plot_placeholder(self):
        """初始化右侧框架中的占位图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, ax in enumerate(axes.flatten()):
            ax.text(0.5, 0.5, f"Placeholder {i+1}", fontsize=14, ha="center", va="center")
            ax.axis("off")

        # 清空右侧框架并嵌入占位图
        self.update_canvas(fig)

        return fig, "Default_Placeholder"

    def update_canvas(self, fig):
        """更新右侧框架中的图像"""
        # 清除旧的 Canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # 嵌入新的图像
        self.canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def plot_results(self, x, y_true_, y_pred_, depthmin, depthmax, model_file):
        """绘制预测结果图像，更新4个子图"""
        y_true_ = y_true_.flatten()
        y_pred_ = y_pred_.flatten()

        # Denormalize
        y_true = y_true_ * (depthmax - depthmin) + depthmin
        y_pred = y_pred_ * (depthmax - depthmin) + depthmin

        # 计算回归直线
        coefficients = np.polyfit(y_true, y_pred, 1)
        poly_eq = np.poly1d(coefficients)
        y_fit = poly_eq(y_true)

        r_squared = r2_score(y_true, y_fit)
        rmse = np.sqrt(mean_squared_error(y_true, y_fit))

        # 创建4个子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 子图 1, 1: 加载的图像 (示例)
        # loaded_image = np.random.random((100, 100))  # 示例加载图像
        loaded_image = x.reshape(rows, cols, -1)
        # 根据图像最后一维度画出调整进度条，每个进度条一格代表一格波段 [rows, cols, 1]
        axes[0, 0].imshow(loaded_image[:, :, 0], cmap='gray', alpha=0.5)
        axes[0, 0].set_title(f"Loaded Image - Band {0+1}")
        axes[0, 0].axis("off")
        # 在axes[0, 0]绘制一个手动可调节的进度条，用于调节波段 loaded_image[:, :, i]

        # 创建一个新的轴用于放置滑块
        slider_ax = fig.add_axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        band_slider = Slider(slider_ax, 'Band', 1, loaded_image.shape[2], valinit=1, valstep=1)

        def update(val):
            band = int(band_slider.val) - 1
            axes[0, 0].imshow(loaded_image[:, :, band], cmap='gray', alpha=0.5)
            axes[0, 0].set_title(f"Loaded Image - Band {band + 1}")
            fig.canvas.draw_idle()

        band_slider.on_changed(update)


        # axes[0, 0].imshow(loaded_image, cmap='gray')
        # axes[0, 0].set_title("Loaded Image")
        # axes[0, 0].axis("off")

        # 子图 1, 2: 深度的图像 (示例)
        # depth_image = np.random.random((100, 100))  # 示例深度图像
        depth_image = y_true.reshape(rows, cols)
        axes[0, 1].imshow(depth_image, cmap='gray')
        axes[0, 1].set_title("Actual Depth Image")
        axes[0, 1].axis("off")

        # 子图 2, 1: 回归曲线
        axes[1, 0].scatter(y_true, y_pred, color='dodgerblue', alpha=0.7, label='Data points')
        axes[1, 0].plot(y_true, y_fit, color='red', linestyle='--', label=f'Fit line ($R^2$={r_squared:.2f}, RMSE={rmse:.2f}m)')
        axes[1, 0].set_xlim(depthmin, depthmax)
        axes[1, 0].set_ylim(depthmin, depthmax)
        axes[1, 0].set_title("Regression Plot")
        axes[1, 0].set_xlabel("Actual Depth (m)")
        axes[1, 0].set_ylabel("Predicted Depth (m)")
        axes[1, 0].legend()

        # 子图 2, 2: 生成的深度图像 (示例)
        # depth_image = np.random.random((100, 100))  # 示例深度图像
        depth_image = y_pred.reshape(rows, cols)
        axes[1, 1].imshow(depth_image, cmap='gray')
        axes[1, 1].set_title("Predicted Depth Image")
        axes[1, 1].axis("off")

        # 更新保存按钮
        modelname = model_file.split('/')[-1].split('.')[0]
        save_button = self.left_frame.winfo_children()[-1]
        save_button.config(command=lambda: self.save_plot(fig, modelname))

        # 更新画布
        self.update_canvas(fig)


    def save_plot(self, fig, modelname):
        """保存图像"""
        if not os.path.exists(f'./results'):
            os.makedirs(f'./results')
        file_path = filedialog.asksaveasfilename(filetypes=[(f'PNG FILE', "*.png")], defaultextension=".png", initialdir=f'./results', initialfile=f'{modelname}.png')
        if file_path:
            fig.savefig(file_path)
            # messagebox.showinfo("Plot Saved", f"Plot saved to: {file_path}")
            print(f'Plot saved to: {file_path}')


# Main application loop
if __name__ == "__main__":
    root = tk.Tk()
    app = SpectralApp(root)
    root.mainloop()
