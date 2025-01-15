#include <iostream>

using namespace std;

int main() {
  int n, t, l;
  cin >> n >> t;
  string input;
  cin >> input;
  l = input.length() - 1;
  while (t--) {
    for (int i = 0; (i < l); i++)
      if (input[i] == 'B' && input[i + 1] == 'G')
        swap(input[i], input[i + 1]), i++;
  }
  cout << input << endl;
  return 0;
}
