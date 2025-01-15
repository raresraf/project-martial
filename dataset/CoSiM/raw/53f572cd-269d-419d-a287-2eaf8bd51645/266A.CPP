#include <algorithm>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string a;
  int n;
  cin >> n;
  cin >> a;
  int count = 0;
  for (int i = 1; i < n; i++) {
    if (a[i] == a[i - 1]) {
      count++;
    }
  }
  cout << count;
  return 0;
}
