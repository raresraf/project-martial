#include <iostream>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  int arr[n];
  int temp;
  int count = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }
  temp = arr[k - 1];
  // cout<<endl<<temp<<endl;
  for (int i = 0; i < n; i++) {
    if (arr[i] >= temp && arr[i] > 0) {
      count++;
    }
  }
  cout << count << endl;

  return 0;
}