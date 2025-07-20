#include <iostream>
#include <vector>
using namespace std;

int check(long long x) {
  if (x < 0)
    return -1;
  else
    return 1;
}
int main() {
  int t;
  cin >> t;
  while (t--) {
    int n;
    cin >> n;
    long long value;
    long long sum = 0;
    vector<long long> arr;
    for (int i = 0; i < n; i++) {
      cin >> value;
      arr.push_back(value);
    }
    for (int i = 0; i < n; i++) {
      int j = i;
      long long maxE = (long long)-1e9;
      while (j < n) {
        if (check(arr[j]) == check(arr[i])) {
          maxE = (maxE < arr[j]) ? arr[j] : maxE;
        } else {
          break;
        }
        j++;
      }
      sum = sum + maxE;
      i = j - 1;
    }
    cout << sum << endl;
  }
}