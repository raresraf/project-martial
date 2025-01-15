// #include <iostream>
// using namespace std;

// int min_moves(int matrix[6][6])
// {
//     int r = 0, c = 0;
//     bool flag = false;
//     for (int i = 1; i <= 5; i++)
//     {
//         for (int j = 1; j <= 5; j++)
//         {
//             if (matrix[i][j] == 1)
//             {
//                 r = i;
//                 c = j;
//                 break;
//                 flag = true;
//             }
//         }
//         if(flag)
//             break;
//     }
//     return abs(r-3)+abs(c-3);
// }

// int main()
// {
//     int matrix[6][6];
//     for (int i = 1; i <= 5; i++)
//         for (int j = 1; j <= 5; j++)
//             cin >> matrix[i][j];

//     cout << min_moves(matrix);
// }

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/

int main() {
  int N;
  cin >> N;
  cin.ignore();
  for (int i = 0; i < N; i++) {
    int X;
    cin >> X;
    cin.ignore();
    string s = "";
    if (X == 0) {
      cout << X;
    } else {
      while (X) {

        s += X % 2;
        X = X / 2;
      }

      for (int i = s.length() - 1; i >= 0; i--) {
        cout << s[i];
      }
    }
    cout << endl;
  }

  // Write an answer using cout. DON'T FORGET THE "<< endl"
  // To debug: cerr << "Debug messages..." << endl;

  // cout << "ones and zeros" << endl;
}