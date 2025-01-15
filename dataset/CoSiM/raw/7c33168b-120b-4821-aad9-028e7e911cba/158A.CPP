#include <iostream>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(false);
  lli a, b, count = 0;
  cin >> a >> b;
  lli arr[a];
  for (int i = 0; i < a; ++i) {
    cin >> arr[i];
  }
  if (arr[0] <= 0) {
    cout << 0 << endl;
    return 0;
  }
  if (arr[b - 1] == 0) {
    for (int i = 0; i < b - 1; ++i) {
      if (arr[i] > 0) {
        count++;
      }
    }
    cout << count << endl;
    return 0;
  }
  for (int i = b; i < a; ++i) {
    if (arr[i] == arr[b - 1] && arr[i] > 0) {
      count++;
    }
  }
  cout << b + count << endl;
  return 0;
}
