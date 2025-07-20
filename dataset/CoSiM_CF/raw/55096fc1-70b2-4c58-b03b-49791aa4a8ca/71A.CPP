#include <iostream>
using namespace std;

void shorten(string str) // Function to abbreviate
{
  int count = 0;

  for (int i = 1; i < str.length() - 1; i++)
    count++;

  cout << str[0] << count << str[str.length() - 1] << endl;
}

int main() {
  int n;
  string str;
  cin >> n;
  getchar(); // To clear input buffer

  for (int i = 0; i < n; i++) {
    getline(cin, str);
    if (str.length() > 10)
      shorten(str);
    else
      cout << str << endl;
  }
  return 0;
}
