#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  bool check = false;

  for (int i = 2; i < n; i = i + 2) {
    if ((n - i) % 2 == 0) {
      cout << "YES";
      check = true;
      break;
    }
  }
  if (!check) {
    cout << "NO";
  }

  return 0;
}