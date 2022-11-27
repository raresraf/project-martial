#include <iostream>
using namespace std;
int main() {
  int flag1 = 0;
  int weight;
  cin >> weight;
  int x, y;
  for (x = 2; x < weight; x = x + 2) {
    y = weight - x;
    if (y % 2 == 0 && y > 0) {
      flag1 = 1;
    } else {
      flag1 = 0;
    }
  }
  if (flag1 == 1) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
