#include <iostream>
using namespace std;

int main() {
  const string hello = "hello";
  string input;
  cin >> input;

  int a = 0, count = 0;
  for (int i = 0; i < input.length(); i++)
    if (input[i] == hello[a]) {
      a++;
      count++;
    }

  if (count == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}