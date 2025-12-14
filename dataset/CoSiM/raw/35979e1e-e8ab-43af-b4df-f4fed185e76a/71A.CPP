#include <cstring>
#include <iostream>

using namespace std;

int main() {
  int n;
  char c[101];
  cin >> n;
  while (n) {
    cin >> c;
    int len = strlen(c);
    if (len > 10) {
      cout << c[0] << len - 2 << c[len - 1] << endl;
    } else {
      cout << c << endl;
    }
    n--;
  }
  return 0;
}