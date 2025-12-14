#include <iostream>
using namespace std;

int main() {
  int i, n, k, flag, p, v[3000];
  cin >> n;
  for (i = 0; i < n * 3; i++)
    cin >> v[i];
  for (i = 0, k = 0, p = 0; i < n * 3; i += 3) {
    flag = 0;
    for (k = i; k < i + 3; k++) {
      if (v[k] == 1)
        flag++;
    }
    if (flag >= 2)
      p++;
  }
  cout << p;
  return 0;
}
