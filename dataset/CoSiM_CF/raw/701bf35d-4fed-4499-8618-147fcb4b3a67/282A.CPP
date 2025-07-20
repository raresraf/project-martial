#include <iostream>
using namespace std;
int main() {
  int x = 0, n;
  string o;
  cin >> n;
  while (n--) {
    cin >> o;
    if (o[1] == '+') {
      x++;
    } else {
      x--;
    }
  }
  cout << x << endl;
}
