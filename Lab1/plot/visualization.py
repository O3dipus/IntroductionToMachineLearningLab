import matplotlib.pyplot as plt


# calculate the width of the tree, i.e. number of leaves
# input :
#       root : node - DecisionTreeNode
# return :
#       width : float
def calc_width(root):
    if root.is_leaf:
        return 1.0
    return calc_width(root.left) + calc_width(root.right)


# plot the image of the decision tree
# input :
#       tree : tree - DecisionTree
#       index : iteration number - int
# return :
#       None
def visualize(tree, index):
    plt.figure(dpi=1200)
    # Remove tick labels and frame
    ax_props = dict(xticks=[], yticks=[])
    visualize.ax = plt.subplot(frameon=False, **ax_props)

    node = tree.root
    total_width = calc_width(node)
    total_depth = float(tree.max_depth)

    # Initial position
    x_pos = -0.5 / total_width
    y_pos = 1.0
    plot_tree = PlotTree(node, (0.5, 1.0), total_width, total_depth, x_pos, y_pos)
    plot_tree.plot()
    plt.savefig('output_plots/noisy_tree_' + str(index) + '.png')


class PlotTree:
    def __init__(self, node, parent_pos, total_width, total_depth, x_pos, y_pos):
        self.node = node
        self.parent_pos = parent_pos
        self.total_width = total_width
        self.total_depth = total_depth
        self.x_pos = x_pos
        self.y_pos = y_pos

    def plot(self):
        width = calc_width(self.node)
        position = (self.x_pos + (1.0 + width) / (2.0 * self.total_width), self.y_pos)
        message = 'x[' + str(self.node.dim) + '] < ' + str(self.node.value)
        self.plot_node(message, position, self.parent_pos)
        self.y_pos -= 1.0 / self.total_depth
        left = self.node.left
        right = self.node.right
        if left:
            if left.is_leaf:
                self.x_pos += 1.0 / self.total_width
                message = 'leaf: ' + str(left.value)
                self.plot_node(message, (self.x_pos, self.y_pos), position)
            else:
                self.node = left
                self.parent_pos = position
                self.plot()

        if right:
            if right.is_leaf:
                self.x_pos += 1.0 / self.total_width
                message = 'leaf: ' + str(left.value)
                self.plot_node(message, (self.x_pos, self.y_pos), position)
            else:
                self.node = right
                self.parent_pos = position
                self.plot()

        self.y_pos += 1.0 / self.total_depth

    # plot tree node
    # input :
    #       message: message on the node - str
    #       position: position of current node/leaf - (float, float)
    #       parent_position: position of parent node - (float, float)
    # return :
    #       None
    def plot_node(self, message, position, parent_position):
        visualize.ax.annotate(message, xy=(parent_position[0], parent_position[1]), xycoords='data',
                              xytext=position, textcoords='data', va="center", ha="center", size=0.5 / self.total_depth,
                              bbox=dict(boxstyle="round", fc='w', ec='b', lw=0.5 / self.total_depth),
                              arrowprops=dict(arrowstyle="-", lw=0.5 / self.total_depth))

