
# We limit processing to the first minute of the video
# should take < 1 minute
f = 'RRR_tutorial.mp4'

cap = cv2.VideoCapture(f)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

stop = int(fps*60) # 1 minute of video
print(f"Stopping after {stop} frames.")

#vid_cod = cv2.VideoWriter_fourcc(*'H264')
vid_cod = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('./ssd_video_mp4v.mp4', vid_cod, int(cap.get(cv2.CAP_PROP_FPS)),(w, h))

cur_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cur_frame = cur_frame+1
    if cur_frame > stop: break #capture first minute

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_im = detectobjects(frame)
    annotated_im = cv2.cvtColor(annotated_im, cv2.COLOR_RGB2BGR)
    out.write(annotated_im)

cap.release()
out.release()

# should take about 20 seconds
# !ffmpeg -y -i ./ssd_video_mp4v.mp4 -vcodec libx264 -f mp4 ./ssd_video.mp4