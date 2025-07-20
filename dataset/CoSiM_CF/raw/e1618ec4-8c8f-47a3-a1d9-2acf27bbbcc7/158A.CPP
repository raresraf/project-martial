#include <iostream>
using namespace std;
int main() {
  int a, b;
  cin >> a >> b;
  int arr[a];
  int t = 0;
  for (int i = 0; i < a; i++) {
    cin >> arr[i];
  }
  for (int i = 0; i < a; i++) {
    if ((arr[b - 1] <= arr[i]) && (arr[i] > 0))
      t++;
  }

  cout << t << endl;
  return 0;
}