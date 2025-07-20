#include <cstring>
#include <iostream>

using namespace std;

int main() {
  int sum = 0, x = 0;
  string com;

  cin >> x;

  for (int i = 0; i < x; i++) {
    cin >> com;
    if (com[1] == '+') {
      sum++;
    } else {
      sum--;
    }
  }
  cout << sum;
}
