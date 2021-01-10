import win32com.client
import win32api
import win32con
import time

VK_CODE = {
	'spacebar':0x20,
	'down_arrow':0x28,
}

class PPTControler:
	def __init__(self):
		# 多线程时会出问题，http://www.cnblogs.com/AlgorithmDot/p/3386972.html
		self.app = win32com.client.Dispatch("PowerPoint.Application")
		self.isFullScreen = False
	def fullScreen(self):
		if not self.isFullScreen:
			#全屏播放
			if self.hasActivePresentation():
				self.app.ActivePresentation.SlideShowSettings.Run()
				self.isFullScreen = True
				return self.getActivePresentationSlideIndex()

	def exitFullScreen(self):
		try:
			if self.app.SlideShowWindows(1).IsFullScreen:
				self.isFullScreen = True
		except:
			self.isFullScreen = False
		if self.isFullScreen:
			if self.hasActivePresentation():
				try:
					if self.app.SlideShowWindows(1).IsFullScreen: ##ppShowTypeKiosk
						self.app.SlideShowWindows(1).View.Exit()
				except:
					self.app.ActiveWindow.Exit()
			self.isFullScreen = False
			
	def activateLaserPointer(self):
		try:
			self.app.SlideShowWindows(1).View.LaserPointerEnabled = True
		except:
			pass

	def disableLaserPointer(self):
		try:
			self.app.SlideShowWindows(1).View.LaserPointerEnabled = False
		except:
			pass
	def click(self):
		win32api.keybd_event(VK_CODE['spacebar'],0,0,0)
		win32api.keybd_event(VK_CODE['spacebar'],0,win32con.KEYEVENTF_KEYUP,0)
		return self.getActivePresentationSlideIndex()

	def gotoSlide(self,index):
		#跳转到指定的页面
		if self.hasActivePresentation():
			try:
				self.app.ActiveWindow.View.GotoSlide(index)
				return self.app.ActiveWindow.View.Slide.SlideIndex
			except:
				self.app.SlideShowWindows(1).View.GotoSlide(index)
				return self.app.SlideShowWindows(1).View.CurrentShowPosition

	def nextPage(self):
		if self.hasActivePresentation():
			count = self.getActivePresentationSlideCount()
			index = self.getActivePresentationSlideIndex()
			return index if index >= count else self.gotoSlide(index+1)

	def prePage(self):
		if self.hasActivePresentation():
			index =  self.getActivePresentationSlideIndex()
			return index if index <= 1 else self.gotoSlide(index-1)

	def getActivePresentationSlideIndex(self):
		#得到活跃状态的PPT当前的页数
		if self.hasActivePresentation():
			try:
				index = self.app.ActiveWindow.View.Slide.SlideIndex
			except:
				index = self.app.SlideShowWindows(1).View.CurrentShowPosition
		return index

	def getActivePresentationSlideCount(self):
		#返回处于活跃状态的PPT的页面总数
		return self.app.ActivePresentation.Slides.Count

	def getPresentationCount(self):
		#返回打开的PPT数目
		return self.app.Presentations.Count

	def hasActivePresentation(self):
		#判断是否有打开PPT文件
		return True if self.getPresentationCount() > 0 else False
if __name__ == '__main__':
	ppt = PPTControler()
	# ppt.fullScreen()
	# for i in range(5):
	# 	time.sleep(1)
	# 	ppt.nextPage()
	# ppt.exitFullScreen()
	# ppt.disableLaserPointer()
	print(len(win32api.EnumDisplayMonitors(None, None)))
