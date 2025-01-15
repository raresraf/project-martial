#include <iostream>

using namespace std;

int main() {
  int k, n;
  cin >> k >> n;
  int a[k + 1];
  for (int i = 0; i < k; i++)
    cin >> a[i];
  a[k] = 0;
  int i = 0;

  while (a[i] >= a[i + 1] && n > 0) {
    if (a[i] == 0)
      break;
    n--;
    if (n == 0) {
      while (a[i] == a[i + 1])
        i++;
    }
    i++;
  }
  cout << i << endl;
  return 0;
}