#include <iostream>
using namespace std;
int main() {
  int x, n1, n2, temp1 = 0, temp2 = 0;
  cin >> x;
  while (x > 0) {
    cin >> n1 >> n2;
    temp1 = n2 - n1 + temp1;
    if (temp1 > temp2) {
      temp2 = temp1;
    }
    x--;
  }
  cout << temp2 << endl;
  return 0;
}
