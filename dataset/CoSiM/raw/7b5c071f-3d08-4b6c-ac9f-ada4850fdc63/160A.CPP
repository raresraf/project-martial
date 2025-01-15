#include <algorithm>
#include <iostream>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(false);
  lli t;
  cin >> t;
  lli arr[t], sum = 0, ms = 0, n = 0;
  for (int i = 0; i < t; ++i) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + t);
  for (int i = t - 1; i >= 0; i--) {
    sum -= arr[i];
    ms += arr[i];
    n++;
    if (ms > sum) {
      cout << n << endl;
      break;
    }
  }
  return 0;
}
