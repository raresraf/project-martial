#include <iostream>
using namespace std;
int main() {

  int n, m;
  string arr;
  cin >> n >> m;
  cin >> arr;
  for (int i = 0; i < m; i++) {
    for (int j = arr.length() - 1; j > 0; j--) {
      if (arr[j] == 'G' && arr[j - 1] == 'B') {
        arr[j] = 'B';
        arr[j - 1] = 'G';
        j--;
      }
    }
  }

  cout << arr;

  return 0;
}
