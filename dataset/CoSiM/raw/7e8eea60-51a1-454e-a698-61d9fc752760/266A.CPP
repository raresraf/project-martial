#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  char stones[n];
  cin >> stones[0];
  int count = 0;
  for (int i = 1; i < n; i++) {
    cin >> stones[i];
    if (stones[i] == stones[i - 1])
      count++;
  }
  cout << count << endl;
  return 0;
}
