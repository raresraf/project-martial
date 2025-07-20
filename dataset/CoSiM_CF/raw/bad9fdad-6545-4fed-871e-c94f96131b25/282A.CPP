#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int n;
  int count = 0;
  cin >> n;
  while (n--) {
    char c[4];
    cin >> c;
    for (int i = 0; i < strlen(c); i++) {
      if (c[i] == '+') {
        count++;
        break;
      } else if (c[i] == '-') {
        count--;
        break;
      }
    }
  }
  cout << count;
  return 0;
}