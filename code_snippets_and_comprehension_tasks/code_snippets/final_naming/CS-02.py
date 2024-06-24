class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

    def __repr__(self):
        string_rep = []
        temp = self
        while temp:
            string_rep.append(f"{temp.data}")
            temp = temp.next
        return "->".join(string_rep)


def make_linked_list(elements_list: list):
    if not elements_list:
        raise Exception("The Elements List is empty")

    current = head = Node(elements_list[0])
    for i in range(1, len(elements_list)):
        current.next = Node(elements_list[i])
        current = current.next
    return head


def print_reverse(head_node: Node) -> None:
    if head_node is not None and isinstance(head_node, Node):
        print_reverse(head_node.next)
        print(head_node.data)