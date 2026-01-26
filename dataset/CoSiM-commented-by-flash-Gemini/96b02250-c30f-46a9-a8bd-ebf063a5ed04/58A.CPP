#include <iostream>
using namespace std;
int main() {
  int i, j = 10, k = 0, n = 0;
  char a[101], b[] = "hello";
  cin >> a;
  for (i = 0; a[i] != '\0'; i++) {
    if (a[i] == b[n]) {
      k++;
      n++;
    }
    if (k == 5) {
      cout << "YES";
      return 0;
    }
  }
  if (k != 5)
    cout << "NO";
  return 0;
}
