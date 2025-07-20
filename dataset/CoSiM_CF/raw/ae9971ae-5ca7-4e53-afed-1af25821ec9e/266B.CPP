#include <algorithm>
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n, i;
  cin >> n >> i;
  string queue;
  cin >> queue;
  for (int t = i; t > 0; t--) {
    string temp = queue;
    for (int j = n - 2; j >= 0; j--) {
      if (queue[j] == 'B' && queue[j + 1] == 'G') {
        swap(temp[j], temp[j + 1]);
      }
    }
    queue = temp;
  }
  cout << queue << endl;
  return 0;
}