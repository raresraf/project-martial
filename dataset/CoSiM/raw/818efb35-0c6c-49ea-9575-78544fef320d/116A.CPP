#include <iostream>
#include <vector>
using namespace std;
int main() {
  int n, max = 0, carry = 0;
  cin >> n;
  vector<int> a(n);
  vector<int> d(n);
  for (int i = 0; i < n; i++)
    cin >> a[i] >> d[i];
  for (int i = 0; i < n; i++) {
    carry = d[i] - a[i] + carry;
    if (max < carry)
      max = carry;
  }
  cout << max << endl;
  return 0;
}