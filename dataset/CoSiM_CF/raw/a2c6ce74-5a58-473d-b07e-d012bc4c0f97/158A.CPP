#include <iostream>
#include <vector>
using namespace std;

int main() {
  int n, k, count = 0;
  cin >> n >> k;
  vector<int> a(n);

  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  for (int i = 0; i < n; i++) {

    if (a.at(i) >= a.at(k - 1) && (a.at(i) != 0 || a.at(k - 1) != 0)) {
      count++;
    } else if (a.at(i) < a.at(k - 1)) {
      break;
    }
  }
  cout << count << endl;
  return 0;
}
