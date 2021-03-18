import torch
import torchvision as tv
import numpy as np

import matplotlib
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt
import gi 
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
class ImshowWindow:
  def __init__(self, data, model, *args):
    self.data = data
    self.fig, self.ax = plt.subplots()
    manager = self.fig.canvas.manager
    self.grid_data = self.asgrid(data)

    self.ax.imshow(self.grid_data)
    toolbar = manager.toolbar
    vbox = manager.vbox



    self.gen = lambda x:  dl.generate_new_subset(
      [random.randint(0,59999) for i in range(int(x))], 
        dl.datamain, dl.targets)



    label_int_cursor = Gtk.Label()
    label_int_cursor.set_markup("values goes here")
    label_int_cursor.show()

    vbox.pack_start(label_int_cursor, False, False, 0)
    vbox.reorder_child(toolbar,-1)


    lbl_row = Gtk.Label()
    lbl_row.set_markup("value row minus ideally")
    lbl_row.show()
    vbox.pack_start(lbl_row, False, False, 0)
    vbox.reorder_child(toolbar,-1)
    
    lbl_pred = Gtk.Label()
    lbl_pred.set_markup("model prediction goes here")
    lbl_row.show()
    vbox.pack_start(lbl_pred, False, False, 0)
    vbox.reorder_child(toolbar,-1)
   
    for arg in args:
      l = Gtk.Label()
      l.set_markup(arg)
      l.show()
      vbox.pack_start(l, False, False, 0)
    
    ax = self.ax
    fig = self.fig
    def loadgrid(self, other=None):
      other.ax.imshow(other.grid_data)
      other.fig.canvas.draw()

    b2grid = Gtk.Button(label = "return to gridview")
    b2grid.show()
    b2grid.connect("clicked", loadgrid, self)
    toolitem = Gtk.ToolItem()
    toolitem.set_tooltip_text("when a clicked image is loaded, clicking this will return to grid view")
    toolitem.show()
    toolitem.add(b2grid)
    toolbar.insert(toolitem, 8)

    generate_set_entry = Gtk.Entry()
    def loadnew(self, entry, other):
      btmp = buf = entry.get_buffer()
      try: buf = int(buf)
      except: entry.set_text(f"must be an integer value, was {btmp}")
      other.loadgrid(other)

    generate_set_entry.show()
    vbox.pack_start(generate_set_entry, False, False, 0)
    vbox.reorder_child(toolbar, -1)
    btn_gen = Gtk.Button(label = "load data of size entered in textfield")
    btn_gen.show()
    btn_gen.connect("clicked", loadnew, generate_set_entry, self)
    vbox.pack_start(btn_gen, False, False, 0) 
    vbox.reorder_child(toolbar, -1)

    self.xvalue = self.yvalue = None
    def update(event):
      if event.xdata is None:
        label_int_cursor.set_markup("values go here right here")
      else:
        label_int_cursor.set_markup(f'<span color="#ef0000">x,y = ({int(event.xdata)}, {int(event.ydata)})</span>')
    self.fig.canvas.mpl_connect("motion_notify_event", update)
    
    def load_img(event):
      if event.xdata is None or event.ydata is None:
        lbl_row.set_markup("lbl_row")
      else:
        self.xvalue = (int(event.xdata) - int(event.xdata//28)*2) // 28
        self.yvalue = (int(event.ydata) - int(event.ydata//28)*2) // 28
        lbl_row.set_markup(f"xvalue is {self.xvalue}, yvalue is {self.yvalue}")
    self.fig.canvas.mpl_connect("motion_notify_event", load_img)


    def zoom_pic(event):
      print(f"this is the raw indexing! {self.data[self.xvalue + self.yvalue* 8]}")
      to_numpy = (self.data[self.xvalue+ self.yvalue *8 ] * 255).to(dtype = torch.uint8).numpy()
      to_numpy = to_numpy.squeeze()
      print(f"This is after numpy conversion {to_numpy}")
      self.ax.imshow(to_numpy, cmap = "gray")
      self.fig.canvas.draw()
    self.fig.canvas.mpl_connect("button_press_event", zoom_pic)





    entry = Gtk.Entry() 
    entry_b = Gtk.Button(label = "loads new subset")
    def loadnew(self, entry, other):  
      b = entry.get_buffer()
      other.data, other.targets = self.gen(int(b))
      other.out = other.model(other.data) 
    entry.show()
    entry_b.show()
    entry_b.connect("clicked", loadnew, entry, self)

  def show(self):
    plt.show()
  def asgrid(self, tensor, rows = 8):
    grid = tv.utils.make_grid(tensor, nrow = rows, pad_value = 0.5)
    try: return grid.numpy().transpose(1,2,0)
    except TypeError: return grid.numpy()




