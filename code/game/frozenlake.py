import gymnasium as gym
import tkinter as tk
from PIL import Image, ImageTk
import pygame

# 創建FrozenLake環境
env = gym.make("FrozenLake-v1", map_name="4x4", render_mode="rgb_array", is_slippery=False)
state = env.reset()[0]

# 初始化步數計數
steps = 0

# 初始化pygame並載入音效
pygame.mixer.init()
fail_sound = pygame.mixer.Sound("water-splash.mp3")  # 失敗音效
success_sound = pygame.mixer.Sound("success_sound.mp3")  # 成功音效


# 建立Tkinter介面
window = tk.Tk()
window.title("Frozen Lake GUI")
window.geometry("400x450")  # 增加高度來顯示步數

# 設置畫布來顯示地圖
canvas = tk.Canvas(window, width=400, height=400)
canvas.pack()

# 步數標籤
step_label = tk.Label(window, text=f"Steps: {steps}")
step_label.pack()

# 渲染函數
def render():
    img = env.render()
    img = Image.fromarray(img).resize((400, 400))  # 將影像轉換為PIL格式並調整大小
    photo = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo  # 保持引用，避免被回收
    window.after(100, render)

# 鍵盤控制函數
def key_press(event):
    global state, steps
    action_map = {'Up': 3, 'Down': 1, 'Left': 0, 'Right': 2}
    if event.keysym in action_map:
        action = action_map[event.keysym]
        state, reward, done, _, _ = env.step(action)
        steps += 1  # 每次移動步數+1
        step_label.config(text=f"Steps: {steps}")  # 更新步數顯示
        if done:
            # 如果遊戲結束，檢查是否成功或失敗
            if reward == 1:
                pygame.mixer.Sound.play(success_sound)  # 播放成功音效
            else:
                pygame.mixer.Sound.play(fail_sound)  # 播放失敗音效
            state = env.reset()[0]
            steps = 0  # 重置步數
            step_label.config(text=f"Steps: {steps}")

# 綁定按鍵事件
window.bind("<KeyPress>", key_press)

# 開始渲染畫面
render()

# 運行Tkinter主循環
window.mainloop()


