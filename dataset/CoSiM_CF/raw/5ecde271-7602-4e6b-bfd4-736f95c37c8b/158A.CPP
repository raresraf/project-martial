#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n;
  cin >> k;
  int a[100];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  int c = 0;
  for (int i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] != 0) {
      c++;
    }
  }
  cout << c;
}
