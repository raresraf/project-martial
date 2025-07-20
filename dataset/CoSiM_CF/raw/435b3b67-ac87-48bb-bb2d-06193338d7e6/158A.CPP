#include <iostream>
using namespace std;
int main() {
  int n, k, i, counter = 0;
  int a[103];
  cin >> n >> k;
  for (i = 1; i <= n; i++) {
    cin >> a[i];
  }
  for (i = 1; i <= n; i++) {
    if (a[i] >= a[k] && a[i] > 0) {
      counter++;
    }
  }
  cout << counter << endl;
}
