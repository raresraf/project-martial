#include <iostream>
using namespace std;
int main() {
  char ch[101];
  int i, count = 0, z = 0;
  cin >> ch;
  for (i = 0; ch[i] != '\0'; i++) {
    if (ch[i] == '1') {
      z = 0;
      count++;
      if (count == 7) {
        cout << "YES";
        return 0;
      }
    } else {
      z++;
      count = 0;
      if (z == 7) {
        cout << "YES";
        return 0;
      }
    }
  }
  cout << "NO";

  return 0;
}
