// 158
#include <iostream>
using namespace std;
int main() {
  int n, k;
  int count = 0;
  int arr[105];
  cin >> n >> k;
  for (int i = 1; i <= n; i++)
    cin >> arr[i];
  for (int i = 1; i <= n; i++) {
    if (arr[i] >= arr[k] && arr[i] != 0)
      count++;
    else
      break;
  }
  cout << count;
  return 0;
}
