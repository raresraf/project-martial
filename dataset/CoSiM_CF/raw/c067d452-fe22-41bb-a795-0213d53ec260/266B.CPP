#include <iostream>
#include <string> //for swap
using namespace std;
int main() {
  int childrenNumber;
  cin >> childrenNumber;
  int amountTransform;
  cin >> amountTransform;
  string childrenOrder;
  cin >> childrenOrder;
  string childrenOrderFixed =
      childrenOrder; // aux string for easier fixing (can be avoided)
  for (int i = 0; i < amountTransform; i++) {
    for (int j = 0; j < childrenNumber - 1; j++) {
      // locate Boy-Girl sequence and change it on string's copy
      if (childrenOrder[j] == 'B' && childrenOrder[j + 1] == 'G')
        swap(childrenOrderFixed[j], childrenOrderFixed[j + 1]);
    }
    childrenOrder = childrenOrderFixed;
  }
  cout << childrenOrder;
  return 0;
}
