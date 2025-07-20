#include <iostream>
#include <string>

using namespace std;

int main() {

  int n, k;
  cin >> n >> k;
  int kthScore;
  int result = 0;
  int *arr;
  arr = new int[n];

  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }
  kthScore = arr[k - 1];

  for (int i = 0; i < n; i++) {
    if (arr[i] > 0 && arr[i] >= kthScore)
      result++;
  }

  cout << result << endl;
  return 0;
}
