#include <iostream>
#include <vector>
using namespace std;
int main() {

  int n;
  cin >> n;
  vector<int> arr;
  int mine = 0;
  int all = 0;

  for (int i = 0; i < n; i++) {
    int sv;
    cin >> sv;
    all += sv;
    arr.push_back(sv);
  }

  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      if (arr[i] < arr[j]) {
        swap(arr[i], arr[j]);
      }
    }
  }

  int i = 0;
  while (i < arr.size() && mine <= all) {
    mine += arr[i];
    all -= arr[i];
    i++;
  }

  cout << i;

  return 0;
}
