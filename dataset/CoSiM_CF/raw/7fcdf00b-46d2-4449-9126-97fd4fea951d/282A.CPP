#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, x = 0;
  string kata;

  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> kata;
    if (kata.compare("X++") == 0 || kata.compare("++X") == 0) {
      x++;
    } else if (kata.compare("X--") == 0 || kata.compare("--X") == 0) {
      x--;
    }
  }
  cout << x;
}
