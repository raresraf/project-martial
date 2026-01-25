#include <algorithm>
#include <iostream>

#define MAX 52
using namespace std;

int main() {
  int n, k, arr[MAX];
  cin >> n >> k;
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
  }
  int value = arr[k], count = 0;
  for (int j = 1; j <= n; j++) {
    if (arr[j] >= value && arr[j] > 0)
      count++;
    else
      break;
  }

  cout << count << endl;
  return 0;
}