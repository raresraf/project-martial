#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  string str;
  while (cin >> str) {
    transform(str.begin(), str.end(), str.begin(),
              ::tolower); // Transform it to lowercase first
    vector<char> data(str.begin(), str.end());
    for (int i = 0; i < data.size();) {
      if (data[i] == 'a' || data[i] == 'o' || data[i] == 'y' ||
          data[i] == 'e' || data[i] == 'u' || data[i] == 'i')
        data.erase(data.begin() + i);
      else
        i++;
    }
    for (char c : data)
      cout << '.' << c;
    cout << endl;
  }
  return 0;
}
