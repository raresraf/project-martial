#include <iostream>
using namespace std;
int main() {
  int n, t;
  cin >> n >> t;
  char a[1000];
  for (int i = 0; i < n; i++)
    cin >> a[i];
  for (int i = 0; i < t; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (a[j] == 'B' && a[j + 1] == 'G') {
        char temp = a[j];
        a[j] = a[j + 1];
        a[j + 1] = temp;
        j++;
      }
    }
  }
  for (int i = 0; i < n; i++)
    cout << a[i];
  return 0;
}